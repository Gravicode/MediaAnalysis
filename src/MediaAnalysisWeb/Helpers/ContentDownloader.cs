using HtmlAgilityPack;
using System.Text;

namespace MediaAnalysisWeb.Helpers
{
    public class ContentDownloader
    {
        public static string DownloadContentAsString(string WebUrl)
        {
            try
            {
                var doc = new HtmlWeb().Load(new Uri(WebUrl));
                var nodes = doc.DocumentNode.SelectSingleNode("//body").DescendantsAndSelf();
                //var word = Console.ReadLine().ToLower();
                var sb = new StringBuilder();
                foreach (var node in nodes)
                {
                    if (node.NodeType == HtmlNodeType.Text && node.ParentNode.Name != "script")
                    {
                        sb.Append(node.InnerText.Trim()+" ");
                    }
                }

                return sb.ToString();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return String.Empty;
            }
            

        }
    }
}
